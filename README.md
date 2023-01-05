<div align="center">
  <img src="https://github.com/C4dynamics/C4dynamics/blob/main/tools/C4dynamics.png">
</div>

#### Tsipor Dynamics
#### Algorithms Engineering and Development
****

C4Dynamics (read Tsipor (bird) Dynamics) is an open-source framework for algorithms engineering! 


[![My Skills](https://skillicons.dev/icons?i=python)](https://skillicons.dev)


Welcome to the C4dynamics - an open-source framework for algorithm engineers who work with physical and dynamical systems. 


This framework is designed to help algorithm engineers quickly and efficiently implement and test new algorithms. 

It includes a variety of tools and features to streamline the development process, including:

* ✅ A comprehensive library of common algorithms and data structures!
* Data-point and rigid-body objects
* 6DOF (six degrees of freedom) simulation
* Seekers and sensors
* Save and plot state-vector in one klick
* ✅ A robust testing suite to ensure the reliability and correctness of your implementation!
* ✅ An intuitive API for easily integrating your algorithms into larger systems!
* ✅ Documentation and examples to help you get up and running quickly!


Whether you're a seasoned algorithm engineer or just getting started, this framework has something to offer. Its modular design allows you to easily pick and choose the components you need, and its active community of contributors is always working to improve and expand its capabilities.

So why wait? Start using the C4dynamics today and take your algorithms engineering to the next level!

## Quickstart

See jupyter notebook demonstrations in examples (GitHub repository). 


Define radar with C4dynamics: 

```
import C4dynamics as c4d
rdr = c4d.seekers.dzradar([0, 0, 0], c4d.filters.filtertype.ex_kalman, 50e-3)
```

Define moving target with C4dynamics: 

```
import C4dynamics as c4d
tgt = c4d.datapoint(x = 1000, vx = 100)
```

Define errors to a general-purpose seeker with C4dynamics: 

```
import C4dynamics as c4d
rdr = c4d.seekers.radar(sf = 0.9, bias = 0, noisestd = 1)
```

# Contributors ✨

Thanks goes to these wonderful people ([:hugs:](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/C4dynamics"><img src="https://github.com/C4dynamics/C4dynamics/blob/main/tools/ziv.png" width="100px;" alt="Ziv Meri"/><br /><sub><b>Ziv Meri</b></sub></a><br /><a href="#maintenance-Smartmind12" title="Maintenance">🚧</a></td>
      <td align="center"><a href="http://santoshb.com.np"><img src="https://avatars.githubusercontent.com/u/23402178?v=4?s=100" width="100px;" alt="Santosh Bhandari"/><br /><sub><b>Santosh Bhandari</b></sub></a><br /><a href="https://github.com/amplication/amplication/commits?author=TheLearneer" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/vincenzodomina"><img src="https://avatars.githubusercontent.com/u/54762917?v=4?s=100" width="100px;" alt="Vincenzo Domina"/><br /><sub><b>Vincenzo Domina</b></sub></a><br /><a href="https://github.com/amplication/amplication/commits?author=vincenzodomina" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

























